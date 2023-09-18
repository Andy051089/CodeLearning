package hello;
import java.util.Scanner;
public class Plus {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請輸入低一個數子");
		int num1 = input.nextInt();
		System.out.println("請輸入第二個數字");
		int num2 = input.nextInt();
		System.out.println("請輸入要加減乘除");
		String a = input.next();
		switch(a) {
		case "加":
			System.out.println(num1+num2);
			break;
		case "減":
			System.out.println(num1-num2);
			break;
		case "乘":
			System.out.println(num1*num2);
			break;
		case "除":
			System.out.println(num1/num2);
			break;	
			}
		
	}
}
