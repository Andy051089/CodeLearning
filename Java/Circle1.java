package hello;
import java.util.Scanner;
public class Circle1 {
	public static void main(String[] args) {
		Scanner input = new Scanner(System.in);
			System.out.print("請輸入圓的半徑");
			int r = input.nextInt();
			System.out.println("請出入圓的高");
			int h = input.nextInt();
			double ans = r*r*3.14*h;
			System.out.println("體積為"+ans);
	}
}
