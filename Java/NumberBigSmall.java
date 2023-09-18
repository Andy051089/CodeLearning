package hello;
import java.util.Scanner;
public class NumberBigSmall {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
		System.out.print("請入第一個數字");
		int num1 = input.nextInt();
		System.out.println("請輸入第二個數字");
		int num2 = input.nextInt();
		if(num1==num2) {
			System.out.println("兩個數相等:"+num1+"="+num2);
		}else if(num1<num2) {
			System.out.println("第一個數小於第二個數:"+num1+"<"+num2);
		}else {
			System.out.println("第一個數大於大二個數:"+num1+">"+num2);
		}
	}
}
